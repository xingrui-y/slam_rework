#include "intrinsic_matrix.h"
#include <iostream>

IntrinsicMatrix::IntrinsicMatrix(int cols, int rows, float fx, float fy, float cx, float cy, int level)
    : width(cols), height(rows), fx(fx), fy(fy), cx(cx), cy(cy), invfx(1.0 / fx), invfy(1.0f / fy), level(level)
{
}

IntrinsicMatrix IntrinsicMatrix::scaled(int level) const
{
    float s = 1.0 / (1 << level);
    return IntrinsicMatrix(width * s, height * s, fx * s, fy * s, cx * s, cy * s, level);
}

IntrinsicMatrixPyramid IntrinsicMatrix::build_pyramid() const
{
    return build_pyramid(level);
}

IntrinsicMatrixPyramid IntrinsicMatrix::build_pyramid(int max_level) const
{
    IntrinsicMatrixPyramid pyramid(max_level);
    for (int level = 0; level < max_level; ++level)
    {
        IntrinsicMatrix temp = scaled(level);
        pyramid[level] = std::make_shared<IntrinsicMatrix>(temp);
    }
    return pyramid;
}