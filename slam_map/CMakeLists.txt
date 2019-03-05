project(slam_map)

add_library(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
PRIVATE
    src/dense_mapping.cpp
)

target_include_directories(${PROJECT_NAME}
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    include_directories(${slam_cuda_INCLUDE_DIRS})
    include_directories(${slam_core_INCLUDE_DIRS})
    include_directories(${slam_util_INCLUDE_DIRS})
)

target_link_libraries(${PROJECT_NAME}
PRIVATE
    Eigen3::Eigen
    slam_core
)

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "" FORCE)
