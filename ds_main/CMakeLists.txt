cmake_minimum_required(VERSION 3.1)
project(ds_main CXX)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

add_executable(dense_slam_gui "")

target_sources(dense_slam_gui
PRIVATE
    src/window_manager.cpp
    src/dense_slam_gui.cpp
)

target_include_directories(dense_slam_gui
PUBLIC
    ${GLFW_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(dense_slam_gui
PRIVATE
    OpenGL::GL
    ${GLFW_LIBRARIES}
)