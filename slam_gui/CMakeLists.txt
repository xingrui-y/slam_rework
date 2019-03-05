project(slam_gui)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/opengl_display.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen
    ${OpenGL_LIBRARIES}
    ${Pangolin_LIBRARIES}
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)