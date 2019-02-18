#include "intrinsic_matrix.h"

IntrinsicMatrix::IntrinsicMatrix(int width, int height, float fx, float fy, float cx, float cy)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), width_(width), height_(height)
{
}

IntrinsicMatrix IntrinsicMatrix::get_scaled(float scale)
{
    return *this * scale;
}

IntrinsicMatrix IntrinsicMatrix::operator*(float scale)
{
    return IntrinsicMatrix(width_ * scale, height_ * scale, fx_ * scale, fy_ * scale, cx_ * scale, cy_ * scale);
}

int IntrinsicMatrix::width() const
{
    return width_;
}

int IntrinsicMatrix::height() const
{
    return height_;
}

float IntrinsicMatrix::fx() const
{
    return fx_;
}

float IntrinsicMatrix::fy() const
{
    return fy_;
}

float IntrinsicMatrix::cx() const
{
    return cx_;
}

float IntrinsicMatrix::cy() const
{
    return cy_;
}