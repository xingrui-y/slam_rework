cmake_minimum_required(VERSION 3.10)
project(slam_core LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${slam_cuda_INCLUDE_DIRS}
    ${slam_util_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_sources(${PROJECT_NAME}
PRIVATE
    src/rgbd_image.cpp
    src/dense_tracking.cpp
    src/dense_odometry.cpp
    src/slam_system.cpp
    src/dense_mapping.cpp
    src/point_struct.cpp
    src/bundle_adjuster.cpp
)

target_link_libraries(${PROJECT_NAME}
    slam_util
    slam_cuda
    Eigen3::Eigen
    # g2o_core
    # g2o_stuff
    # g2o_types_sba
    # g2o_solver_dense
    # g2o_solver_cholmod
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)

add_executable(unitest_core "")

target_include_directories(unitest_core
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
PRIVATE
    ${slam_cuda_INCLUDE_DIRS}
    ${slam_util_INCLUDE_DIRS}
)

target_sources(unitest_core
PRIVATE
    test/unit_test.cpp
)

target_link_libraries(unitest_core
    slam_core
    Eigen3::Eigen
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)