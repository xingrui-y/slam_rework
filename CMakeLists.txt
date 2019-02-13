cmake_minimum_required(VERSION 3.4)
project(slam VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(slam_core)
add_subdirectory(slam_app)
add_subdirectory(slam_gui)

add_executable(tum_app src/tum_test.cpp)
target_link_libraries(tum_app slam_core slam_gui slam_app)

add_executable(live_system src/live_system.cpp)
target_link_libraries(live_system slam_core slam_gui slam_app)
