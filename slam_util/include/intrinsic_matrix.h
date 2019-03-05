#ifndef __INTRINSIC_MATRIX__
#define __INTRINSIC_MATRIX__

#include <memory>
#include <vector>

struct IntrinsicMatrix;
using IntrinsicMatrixPtr = std::shared_ptr<IntrinsicMatrix>;
using IntrinsicMatrixPyramid = std::vector<IntrinsicMatrixPtr>;

struct IntrinsicMatrix
{
    IntrinsicMatrix() = default;
    IntrinsicMatrix(const IntrinsicMatrix &) = default;
    IntrinsicMatrix(int cols, int rows, float fx, float fy, float cx, float cy, int level);
    IntrinsicMatrix scaled(int level) const;
    IntrinsicMatrixPyramid build_pyramid() const;
    IntrinsicMatrixPyramid build_pyramid(int max_level) const;

    size_t width, height, level;
    float fx, fy, cx, cy, invfx, invfy;
};

#endif