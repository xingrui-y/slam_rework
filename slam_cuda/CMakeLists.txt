cmake_minimum_required(VERSION 3.10)
project(slam_cuda LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
    src/se3_reduction.cu
    src/map_struct.cu
    src/map_update.cu
    src/map_render.cu
    src/map_export.cu
    src/image_ops.cu
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${slam_util_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME} 
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)

cuda_add_executable(unitest_cuda "")

target_include_directories(unitest_cuda
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_sources(unitest_cuda
PRIVATE
    test/unit_test.cpp
)

set_target_properties(unitest_cuda
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

target_link_libraries(unitest_cuda
    ${OpenCV_LIBRARIES}
    Eigen3::Eigen
    slam_cuda
    slam_util
)