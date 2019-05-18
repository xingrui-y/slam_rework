#include <iostream>
#include "window_manager.h"

int main(int argc, char **argv)
{
    WindowManager wm;

    if (wm.initialize_gl_context())
        wm.render_scene();

    return 0;
}
