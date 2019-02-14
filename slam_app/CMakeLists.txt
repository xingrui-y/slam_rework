project(slam_app)

set(COMMON_SRC 
    src/openni_camera
    src/tum_dataset_wrapper.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${slam_core_INCLUDE_DIRS})
include_directories(${slam_gui_INCLUDE_DIRS})

add_executable(tum_app src/tum_test.cpp ${COMMON_SRC})
target_link_libraries(tum_app slam_core slam_gui)