#ifndef __INTRINSIC_MATRIX__
#define __INTRINSIC_MATRIX__

#include <memory>
#include <vector>

struct IntrinsicMatrix;
typedef std::shared_ptr<IntrinsicMatrix> IntrinsicMatrixPtr;
typedef std::vector<IntrinsicMatrixPtr> IntrinsicMatrixPyramid;

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