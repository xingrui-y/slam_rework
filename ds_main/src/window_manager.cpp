#include "window_manager.h"
#include <iostream>

static void error_callback(int error, const char *description)
{
    std::cerr << description << std::endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

WindowManager::WindowManager() : texture_image(-1)
{
}

WindowManager::~WindowManager()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool WindowManager::initialize_gl_context()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        return false;

    window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);

    return true;
}

void WindowManager::set_image_texture(cv::Mat image)
{
    if (texture_image <= 0)
    {
        glGenTextures(1, &texture_image);
    }

    glBindTexture(GL_TEXTURE_2D, texture_image);
    GLenum colour_format;

    if (image.channels() != 1)
        colour_format = GL_BGR;
    else
        colour_format = GL_LUMINANCE;

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, colour_format, GL_UNSIGNED_BYTE, image.ptr());
}

void WindowManager::render_scene()
{
    while (!glfwWindowShouldClose(window))
    {
        float ratio;
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        ratio = width / (float)height;
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glRotatef((float)glfwGetTime() * 50.f, 0.f, 0.f, 1.f);
        glBegin(GL_TRIANGLES);
        glColor3f(1.f, 0.f, 0.f);
        glVertex3f(-0.6f, -0.4f, 0.f);
        glColor3f(0.f, 1.f, 0.f);
        glVertex3f(0.6f, -0.4f, 0.f);
        glColor3f(0.f, 0.f, 1.f);
        glVertex3f(0.f, 0.6f, 0.f);
        glEnd();
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}