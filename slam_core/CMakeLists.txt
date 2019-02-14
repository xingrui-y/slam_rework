project(slam_core)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/rgbd_image.cpp
    src/dense_tracking.cpp
    src/dense_odometry.cpp
    src/slam_system.cpp
    src/keyframe_graph.cpp
)

target_link_libraries(${PROJECT_NAME} 
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)