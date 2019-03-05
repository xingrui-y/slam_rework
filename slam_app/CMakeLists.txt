project(slam)

set(COMMON_SRC 
    src/program_options.cpp
    src/openni_camera
    src/slam_wrapper
    src/tum_dataset_wrapper.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${slam_core_INCLUDE_DIRS})
include_directories(${slam_gui_INCLUDE_DIRS})
include_directories(${slam_util_INCLUDE_DIRS})
include_directories(${slam_map_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} app/slam_app.cpp ${COMMON_SRC})
target_link_libraries(${PROJECT_NAME} Boost::program_options)
target_link_libraries(${PROJECT_NAME} slam_core slam_gui slam_map)

add_executable(lkt app/lkt_demo.cpp ${COMMON_SRC})
target_link_libraries(lkt Boost::program_options)
target_link_libraries(lkt slam_core slam_gui slam_util slam_map)

