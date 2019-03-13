#include "keyframe_graph.h"
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

class KeyFrameGraph::KeyFrameGraphImpl
{
  public:
    KeyFrameGraphImpl();
    void insert_keyframe(RgbdFramePtr kf);
    void clear_graph();
    void optimise_graph(int iteration);

    g2o::SparseOptimizer graph;
    std::list<RgbdFramePtr> keyframe_list;
    std::queue<RgbdFramePtr> local_frames;
};

KeyFrameGraph::KeyFrameGraphImpl::KeyFrameGraphImpl()
{
    graph.setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
    linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver)));
    graph.setAlgorithm(solver);
}

void KeyFrameGraph::KeyFrameGraphImpl::insert_keyframe(RgbdFramePtr kf)
{
}

void KeyFrameGraph::KeyFrameGraphImpl::clear_graph()
{
    graph.clear();
    graph.clearParameters();
}

void KeyFrameGraph::KeyFrameGraphImpl::optimise_graph(int iteration)
{
    if (graph.edges().size() <= 0)
        return;

    graph.initializeOptimization();
    graph.optimize(iteration, false);
}

KeyFrameGraph::KeyFrameGraph() : impl(new KeyFrameGraphImpl())
{
}