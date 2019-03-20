#include "opengl_display.h"
#include <sophus/se3.hpp>
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <vector>

struct MeshBuffer
{
    MeshBuffer();
    ~MeshBuffer();

    pangolin::GlBuffer *buffer_vertex;
    pangolin::GlBuffer *buffer_normal;
    pangolin::GlBuffer *buffer_texture;

    size_t num_elements;
    const size_t max_elements = 60000000;
};

MeshBuffer::MeshBuffer() : num_elements(0)
{
    buffer_vertex = new pangolin::GlBuffer(pangolin::GlArrayBuffer, max_elements, GL_FLOAT, 3);
    buffer_normal = new pangolin::GlBuffer(pangolin::GlArrayBuffer, max_elements, GL_FLOAT, 3);
    buffer_texture = new pangolin::GlBuffer(pangolin::GlArrayBuffer, max_elements, GL_FLOAT, 3);
}

MeshBuffer::~MeshBuffer()
{
}

class GlDisplay::GlDisplayImpl
{
  public:
    GlDisplayImpl(int width, int height);
    bool should_quit() const;

    void set_camera_pose(const Sophus::SE3d &pose);
    void set_model_view_matrix(const Sophus::SE3d &pose);

    void switch_mesh_buffer();

    void draw_frame();
    void draw_camera(const Sophus::SE3d pose) const;
    void draw_keyframe_graph() const;
    void draw_camera_trajectory() const;
    void draw_ground_truth_trajectory() const;
    void draw_mesh_shaded(pangolin::GlSlProgram *program);

    pangolin::OpenGlRenderState camera;
    pangolin::View model_view_camera;
    pangolin::View main_menu_panel;

    pangolin::Var<bool> *btn_system_reset;
    pangolin::Var<bool> *btn_system_reboot;
    pangolin::Var<bool> *btn_follow_camera;
    pangolin::Var<bool> *btn_show_ground_truth;
    pangolin::Var<bool> *btn_show_camera_trajectory;
    pangolin::Var<bool> *btn_show_shaded_mesh;
    pangolin::Var<bool> *btn_show_current_camera;
    pangolin::Var<bool> *btn_show_keyframe_graph;

    //GLSL shaders
    pangolin::GlSlProgram phong_shader;

    Sophus::SE3d current_pose;
    std::vector<Sophus::SE3d> ground_truth;
    std::vector<Sophus::SE3d> camera_trajectory;
    std::vector<Sophus::SE3d> keyframe_poses;

    std::shared_ptr<MeshBuffer> buffer[2];
    std::mutex mutex_buffer_;
};

GlDisplay::GlDisplayImpl::GlDisplayImpl(int width, int height)
{
    pangolin::CreateWindowAndBind("slam", width, height);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    phong_shader.AddShaderFromFile(pangolin::GlSlVertexShader, "./glsl/phong_shader.glsl");
    phong_shader.AddShaderFromFile(pangolin::GlSlFragmentShader, "./glsl/fragment_shader.glsl");
    phong_shader.Link();

    buffer[0] = std::make_shared<MeshBuffer>();
    buffer[1] = std::make_shared<MeshBuffer>();

    camera = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, 528.f, 528.f, 320.f, 240.f, 0.1f, 100.f),
                                         pangolin::ModelViewLookAtRUB(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f));

    model_view_camera = pangolin::CreateDisplay().SetAspect(-width / (float)height).SetHandler(new pangolin::Handler3D(camera));

    main_menu_panel = pangolin::CreatePanel("MainUI").SetBounds(0, 1.0, 0, pangolin::Attach::Pix(200), true);

    btn_system_reset = new pangolin::Var<bool>("MainUI.Reset System", false, false);
    btn_system_reboot = new pangolin::Var<bool>("MainUI.Reboot System", false, false);
    btn_follow_camera = new pangolin::Var<bool>("MainUI.Follow Camera", false, true);
    btn_show_ground_truth = new pangolin::Var<bool>("MainUI.Show Ground Truth", true, true);
    btn_show_camera_trajectory = new pangolin::Var<bool>("MainUI.Show Camera Trajectory", true, true);
    btn_show_shaded_mesh = new pangolin::Var<bool>("MainUI.Show Mesh Phong", true, true);
    btn_show_current_camera = new pangolin::Var<bool>("MainUI.Show Camera", false, true);
    btn_show_keyframe_graph = new pangolin::Var<bool>("MainUI.Show Key Frame Graph", false, true);
}

void GlDisplay::GlDisplayImpl::switch_mesh_buffer()
{
    std::lock_guard<std::mutex> lock(mutex_buffer_);
    std::swap(buffer[0], buffer[1]);
}

void GlDisplay::GlDisplayImpl::draw_mesh_shaded(pangolin::GlSlProgram *program)
{
    if (buffer[0] && buffer[0]->num_elements != 0)
    {
        std::lock_guard<std::mutex> lock(mutex_buffer_);

        program->SaveBind();
        program->SetUniform("viewMat", camera.GetModelViewMatrix());
        program->SetUniform("projMat", camera.GetProjectionMatrix());
        Eigen::Vector3f translation = current_pose.translation().cast<float>();
        program->SetUniform("lightpos", translation(0), translation(1), translation(2));

        GLuint vao_mesh;
        glGenVertexArrays(1, &vao_mesh);
        glBindVertexArray(vao_mesh);
        buffer[0]->buffer_vertex->Bind();
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(0);
        buffer[0]->buffer_vertex->Unbind();

        buffer[0]->buffer_normal->Bind();
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 0, 0);
        glEnableVertexAttribArray(1);
        buffer[0]->buffer_normal->Unbind();

        glDrawArrays(GL_TRIANGLES, 0, buffer[0]->num_elements * 3);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        program->Unbind();
        glBindVertexArray(0);
    }
}

std::vector<GLfloat> get_camera_wireframe_coord(const Sophus::SE3d pose)
{
    std::vector<GLfloat> transformed_wireframe;
    std::vector<Eigen::Vector3f> wire_frame = {{1, 1, 1}, {1, -1, 1}, {0, 0, 0}, {1, -1, 1}, {-1, -1, 1}, {0, 0, 0}, {-1, -1, 1}, {-1, 1, 1}, {0, 0, 0}, {-1, 1, 1}, {1, 1, 1}, {0, 0, 0}};

    for (auto vertex : wire_frame)
    {
        vertex(1) *= 1.5;
        vertex *= 0.01f;
        vertex = pose.cast<float>() * vertex;
        transformed_wireframe.push_back(vertex(0));
        transformed_wireframe.push_back(vertex(1));
        transformed_wireframe.push_back(vertex(2));
    }

    return transformed_wireframe;
}

void GlDisplay::GlDisplayImpl::draw_camera(const Sophus::SE3d pose) const
{
    auto cam = get_camera_wireframe_coord(pose);

    glColor3f(0.0, 1.0, 0.0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    pangolin::glDrawVertices(cam.size() / 3, (GLfloat *)&cam[0], GL_TRIANGLES, 3);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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

    if (*btn_show_shaded_mesh)
        draw_mesh_shaded(&phong_shader);

    if (*btn_show_keyframe_graph)
        draw_keyframe_graph();

    if (*btn_show_current_camera)
        draw_camera(current_pose);

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

void GlDisplay::GlDisplayImpl::draw_keyframe_graph() const
{
    for (auto pose : keyframe_poses)
    {
        draw_camera(pose);
    }
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

GlDisplay::~GlDisplay()
{
    delete impl->btn_system_reset;
    delete impl->btn_system_reboot;
    delete impl->btn_follow_camera;
    delete impl->btn_show_ground_truth;
    delete impl->btn_show_camera_trajectory;
    delete impl->btn_show_shaded_mesh;
    delete impl->btn_show_current_camera;
    delete impl->btn_show_keyframe_graph;
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

void GlDisplay::set_camera_trajectory(const std::vector<Sophus::SE3d> camera)
{
    impl->camera_trajectory = camera;
}

void GlDisplay::set_keyframe_poses(const std::vector<Sophus::SE3d> keyframes)
{
    impl->keyframe_poses = keyframes;
}

void GlDisplay::upload_mesh(const void *vertices, const void *normal, const void *texture, const size_t &size)
{
    impl->buffer[1]->buffer_vertex->Upload(vertices, size * sizeof(float));
    impl->buffer[1]->buffer_normal->Upload(normal, size * sizeof(float));
    impl->buffer[1]->buffer_texture->Upload(texture, size * sizeof(float));
    impl->buffer[1]->num_elements = size;
    impl->switch_mesh_buffer();
}