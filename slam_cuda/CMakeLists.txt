project(slam_cuda CXX CUDA)

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

#find_package(CUDA 9.2 REQUIRED)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${slam_util_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME} 
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    Eigen3::Eigen
)

target_compile_options(${PROJECT_NAME}
PRIVATE
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)

add_executable(unitest_cuda "")

target_include_directories(unitest_cuda
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

target_sources(unitest_cuda
PRIVATE
    test/unit_test.cpp
)

target_link_libraries(unitest_cuda
    slam_cuda
)