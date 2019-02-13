project(slam_gui)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/pangolin_gui.cpp
)

find_package(Eigen3 3.2 REQUIRED)
find_package(OpenGL 2.0 REQUIRED)
find_package(Pangolin REQUIRED)

target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${OpenGL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES})
