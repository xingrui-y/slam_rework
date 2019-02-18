project(slam_cuda CXX CUDA)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
    src/image_ops.cu
)

find_package(CUDA 9.2 REQUIRED)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(${PROJECT_NAME} 
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
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