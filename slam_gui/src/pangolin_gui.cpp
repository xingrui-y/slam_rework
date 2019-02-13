#include "pangolin_gui.h"
#include <pangolin/pangolin.h>

class PangolinGUI::PangolinGUIImpl
{
  public:
    PangolinGUIImpl(int width, int height);
    void draw_frame();
    bool should_quit() const;

    pangolin::OpenGlRenderState camera;
    pangolin::View model_view_camera;
    pangolin::View main_menu_panel;
    pangolin::Var<bool> *btn_system_reset;
};

PangolinGUI::PangolinGUIImpl::PangolinGUIImpl(int width, int height)
{
    pangolin::CreateWindowAndBind("slam", width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    std::string config_file_path;

    camera = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(width, height, 528.f, 528.f, 320.f, 240.f, 0.1f, 100.f),
                                         pangolin::ModelViewLookAtRUB(0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, -1.f, 0.f));

    model_view_camera = pangolin::CreateDisplay().SetAspect(-width / (float)height).SetHandler(new pangolin::Handler3D(camera));

    main_menu_panel = pangolin::CreatePanel("MainUI").SetBounds(0, 1.0, 0, pangolin::Attach::Pix(200), true);
    btn_system_reset = new pangolin::Var<bool>("MainUI.Reset System", false, false);
}

void PangolinGUI::PangolinGUIImpl::draw_frame()
{
    glClearColor(0.0f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    model_view_camera.Activate(camera);
    pangolin::FinishFrame();
}

bool PangolinGUI::PangolinGUIImpl::should_quit() const
{
    return pangolin::ShouldQuit();
}

PangolinGUI::PangolinGUI(int width, int height)
    : impl(new PangolinGUIImpl(width, height))
{
}

bool PangolinGUI::should_quit() const
{
    return impl->should_quit();
}

void PangolinGUI::draw_frame()
{
    impl->draw_frame();
}