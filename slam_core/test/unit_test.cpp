#include "intrinsic_matrix.h"
#include <cassert>

int main(int argc, char **argv)
{
    // Test the implementation of the intrinsic matrix
    IntrinsicMatrix k(640, 480, 528, 528, 320, 240);
    IntrinsicMatrix k2 = k.get_scaled(0.5);
    assert(k2.width() == 320 && k2.height() == 240);
    assert(k2.fx() == 264 && k2.fy() == 264);
    assert(k2.cx() == 160 && k2.cy() == 120);
}