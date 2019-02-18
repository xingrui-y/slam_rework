#ifndef __INTRINSIC_MATRIX__
#define __INTRINSIC_MATRIX__

class IntrinsicMatrix
{
public:
  IntrinsicMatrix(int width, int height, float fx, float fy, float cx, float cy);
  IntrinsicMatrix get_scaled(float scale);
  IntrinsicMatrix operator*(float scale);

  int width() const;
  int height() const;
  float fx() const;
  float fy() const;
  float cx() const;
  float cy() const;

private:
  int width_, height_;
  float fx_, fy_, cx_, cy_;
};

#endif