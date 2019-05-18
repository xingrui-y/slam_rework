#ifndef __WINDOW_MANAGER__
#define __WINDOW_MANAGER__

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

class WindowManager
{
public:
    WindowManager();
    ~WindowManager();
    bool initialize_gl_context();
    void render_scene();
    void set_image_texture(cv::Mat image);
    void set_slam_system(void *slam);

private:
    GLFWwindow *window;
    GLuint texture_image;
};

#endif