#include "opengl_display.h"
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
#include <vector>

const Eigen::Vector3f cam_vertices[12] =
    {{0.04, 0.03, 0.03},
     {0.04, -0.03, 0.03},
     {0, 0, 0},
     {-0.04, 0.03, 0.03},
     {-0.04, -0.03, 0.03},
     {0, 0, 0},
     {0.04, 0.03, 0.03},
     {-0.04, 0.03, 0.03},
     {0, 0, 0},
     {0.04, -0.03, 0.03},
     {-0.04, -0.03, 0.03},
     {0, 0, 0}};

std::vector<GLfloat> generate_camera_points(const Sophus::SE3d &pose, float scale)
{
    std::vector<GLfloat> result;
    auto r = pose.rotationMatrix().cast<float>();
    auto t = pose.translation().cast<float>();

    for (auto vertex : cam_vertices)
    {
        auto vertex_transformed = r * vertex * scale + t;
        result.push_back(vertex_transformed(0));
        result.push_back(vertex_transformed(1));
        result.push_back(vertex_transformed(2));
    }

    return result;
}

class GlDisplay::GlDisplayImpl
{
  public:
    GlDisplayImpl(int width, int height);
    void draw_frame();
    void draw_camera();
    bool should_quit() const;
    void set_camera_pose(const Sophus::SE3d &pose);
    void set_model_view_matrix(const Sophus::SE3d &pose);
    void draw_camera_trajectory() const;
    void draw_ground_truth_trajectory() const;

    pangolin::OpenGlRenderState camera;
    pangolin::View model_view_camera;
    pangolin::View main_menu_panel;

    pangolin::Var<bool> *btn_system_reset;
    pangolin::Var<bool> *btn_system_reboot;
    pangolin::Var<bool> *btn_follow_camera;
    pangolin::Var<bool> *btn_show_ground_truth;
    pangolin::Var<bool> *btn_show_camera_trajectory;

    Sophus::SE3d current_pose;
    std::vector<Sophus::SE3d> ground_truth;
    std::vector<Sophus::SE3d> camera_trajectory;
};

GlDisplay::GlDisplayImpl::GlDisplayImpl(int width, int height)
{
    pangolin::CreateWindowAndBind("slam", width, height);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    camera = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, 528.f, 528.f, 320.f, 240.f, 0.1f, 100.f),
                                         pangolin::ModelViewLookAtRUB(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f));

    model_view_camera = pangolin::CreateDisplay().SetAspect(-width / (float)height).SetHandler(new pangolin::Handler3D(camera));

    main_menu_panel = pangolin::CreatePanel("MainUI").SetBounds(0, 1.0, 0, pangolin::Attach::Pix(200), true);

    btn_system_reset = new pangolin::Var<bool>("MainUI.Reset System", false, false);
    btn_system_reboot = new pangolin::Var<bool>("MainUI.Reboot System", false, false);
    btn_follow_camera = new pangolin::Var<bool>("MainUI.Follow Camera", false, true);
    btn_show_ground_truth = new pangolin::Var<bool>("MainUI.Show Ground Truth", true, true);
    btn_show_camera_trajectory = new pangolin::Var<bool>("MainUI.Show Camera Trajectory", true, true);
}

void GlDisplay::GlDisplayImpl::draw_camera()
{
    auto cam = generate_camera_points(current_pose, 15);
    GLfloat rgb_active_cam[] = {0.f, 1.f, 0.f};
    glColor3fv(rgb_active_cam);
    pangolin::glDrawVertices(cam.size() / 3, (GLfloat *)&cam[0], GL_LINE_STRIP, 3);
}

void GlDisplay::GlDisplayImpl::draw_frame()
{
    glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    model_view_camera.Activate(camera);

    if (*btn_follow_camera)
        set_model_view_matrix(current_pose);

    if (*btn_show_ground_truth)
        draw_ground_truth_trajectory();

    if (*btn_show_camera_trajectory)
        draw_camera_trajectory();

    draw_camera();
    pangolin::FinishFrame();
}

void GlDisplay::GlDisplayImpl::draw_ground_truth_trajectory() const
{
    std::vector<GLfloat> gt;
    for (int i = 0; i < ground_truth.size(); ++i)
    {
        const Sophus::SE3d &curr = ground_truth[i];
        auto t = curr.translation();
        gt.push_back(t(0));
        gt.push_back(t(1));
        gt.push_back(t(2));
    }

    glColor3f(0.0f, 1.0f, 0.0f);
    pangolin::glDrawVertices(gt.size() / 3, (GLfloat *)&gt[0], GL_LINE_STRIP, 3);
}

void GlDisplay::GlDisplayImpl::draw_camera_trajectory() const
{
    std::vector<GLfloat> camera;
    for (int i = 0; i < camera_trajectory.size(); ++i)
    {
        const Sophus::SE3d &curr = camera_trajectory[i];
        auto t = curr.translation();
        camera.push_back(t(0));
        camera.push_back(t(1));
        camera.push_back(t(2));
    }

    glColor3f(1.0f, 0.0f, 0.0f);
    pangolin::glDrawVertices(camera.size() / 3, (GLfloat *)&camera[0], GL_LINE_STRIP, 3);
}

void GlDisplay::GlDisplayImpl::set_model_view_matrix(const Sophus::SE3d &pose)
{
    Eigen::Vector3d up = {0, -1, 0}, eye = {0, 0, 0}, look = {0, 0, 1};
    // up vector is the up direction of the camera
    up = pose.rotationMatrix() * up;
    // eye point which happens to be the translational part of the camera pose
    eye = pose.rotationMatrix() * eye + pose.translation();
    // looking at : NOTE this is a point in the world coordinate rather than a vector
    look = pose.rotationMatrix() * look + pose.translation();

    // set model view matrix ( eye, look, up ) OpenGl style;
    camera.SetModelViewMatrix(pangolin::ModelViewLookAtRUB(eye(0), eye(1), eye(2),
                                                           look(0), look(1), look(2),
                                                           up(0), up(1), up(2)));
}

bool GlDisplay::GlDisplayImpl::should_quit() const
{
    return pangolin::ShouldQuit();
}

GlDisplay::GlDisplay() : GlDisplay(1280, 960)
{
}

GlDisplay::GlDisplay(int width, int height) : impl(new GlDisplayImpl(width, height))
{
}

bool GlDisplay::should_quit() const
{
    return impl->should_quit();
}

void GlDisplay::draw_frame() const
{
    impl->draw_frame();
}

void GlDisplay::set_current_pose(const Sophus::SE3d &pose) const
{
    impl->current_pose = pose;
}

void GlDisplay::set_ground_truth_trajectory(const std::vector<Sophus::SE3d> &gt)
{
    impl->ground_truth = gt;
}

void GlDisplay::set_camera_trajectory(const std::vector<Sophus::SE3d> &camera)
{
    impl->camera_trajectory = camera;
}